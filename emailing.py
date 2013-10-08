# text me when you're done
def emailWhenDone():
    import smtplib

    to = 'lanemcintosh@gmail.com' #insert reciever email address (can be same as sender)
    gmail_user = 'mcintoshlane@gmail.com' #your gmail sender address
    gmail_pwd = 'hansolo8chewy' #your gmail password
    smtpserver = smtplib.SMTP("smtp.gmail.com",587) #the technical stuff
    smtpserver.ehlo() #the technical stuff
    smtpserver.starttls() #the technical stuff
    smtpserver.ehlo #the technical stuff
    smtpserver.login(gmail_user, gmail_pwd) #the technical stuff
    header = 'To:' + to + '\n' + 'From: ' + gmail_user + '\n' + 'Subject:iPython Notebook \n'
    msg = header + '\n' + 'Your Python Script has now Completed!' #The completion message
    smtpserver.sendmail(gmail_user, to, msg) #Sending the mail
    smtpserver.close() #closing the mailserver connection


# text when error
def emailWhenError(sysexecinfo):
    import traceback
    import smtplib
    
    """ Trace exceptions """
    exc_type, exc_value, exc_traceback = sysexecinfo
    i, j = (traceback.extract_tb(exc_traceback, 1))[0][0:2]
    k = (traceback.format_exception_only(exc_type, exc_value))[0]
    
    ## Text me when you're done
    to = 'lanemcintosh@gmail.com' #insert reciever email address (can be same as sender)
    gmail_user = 'mcintoshlane@gmail.com' #your gmail sender address
    gmail_pwd = 'hansolo8chewy' #your gmail password
    smtpserver = smtplib.SMTP("smtp.gmail.com",587) #the technical stuff
    smtpserver.ehlo() #the technical stuff
    smtpserver.starttls() #the technical stuff
    smtpserver.ehlo #the technical stuff
    smtpserver.login(gmail_user, gmail_pwd) #the technical stuff
    header = 'To:' + to + '\n' + 'From: ' + gmail_user + '\n' + 'Subject:Error in iPython Notebook \n'
    msg = header + '\n' + 'Bummer :(' + '\n' + str(k) #The completion message
    smtpserver.sendmail(gmail_user, to, msg) #Sending the mail
    smtpserver.close() #closing the mailserver connection
    
    
